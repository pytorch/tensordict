# TensorDictBase.exclude

TensorDictBase.exclude(**keys: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Excludes the keys of the tensordict and returns a new tensordict without these entries.

The values are not copied. The result shares its tensors with the
original, so an in-place change to a tensor in one tensordict shows in
the other. Unlike [`select()`](tensordict.TensorDictBase.select.html#tensordict.TensorDictBase.select), a missing key is ignored: there is
no `strict` argument. See [Keys, nested keys, and locking](../../concepts/keys.html#keys).

Parameters:

- ***keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - keys to exclude.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - if `True`, the keys are removed from this
tensordict, and this tensordict is returned.
Default is `False`.

Returns:

A new tensordict (or the same if `inplace=True`) without the excluded entries.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": 0, "b": {"c": 1, "d": 2}}, [])
>>> td.exclude("a", ("b", "c"))
TensorDict(
 fields={
 b: TensorDict(
 fields={
 d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> td.exclude("a", "b")
TensorDict(
 fields={
 },
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> list(td.exclude("this key does not exist").keys())
['a', 'b']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `exclude`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys
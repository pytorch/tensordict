# TensorDictBase.select

TensorDictBase.select(**keys: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *as_tensordict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Selects the keys of the tensordict and returns a new tensordict with only the selected keys.

The values are not copied. The result shares its tensors with the
original, so an in-place change to a tensor in one tensordict shows in
the other. A nested key selects one entry of a sub-tensordict, and a
key of a sub-tensordict selects the whole sub-tensordict. See
[Keys, nested keys, and locking](../../concepts/keys.html#keys).

Parameters:

- ***keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - keys to select.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - if `True`, the other keys are removed from this
tensordict, and this tensordict is returned.
Default is `False`.
- **strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a missing key raises
`KeyError`. If `False`, missing keys are ignored.
Default: `True`.
- **as_tensordict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the result will be a
plain [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) even when called on a
[`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass) instance. This avoids the
TensorClass wrapper that fills unselected fields with `None`.
For [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) subclasses, this is a
no-op. Default: `False`.

Returns:

A new tensordict (or the same if `inplace=True`) with the selected keys only.

Note

To select keys in a tensordict and return a version of this tensordict
deprived of these keys, see the [`split_keys()`](tensordict.TensorDictBase.split_keys.html#tensordict.TensorDictBase.split_keys) method.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": 0, "b": {"c": 1, "d": 2}}, [])
>>> td.select("a", ("b", "c"))
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> td.select("a", "b")
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> td.select("this key does not exist", strict=False)
TensorDict(
 fields={
 },
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `select`

![](../../_images/sphx_glr_streamed_tensordict_thumb.png)

[Building tensordicts from streams](../../tutorials/streamed_tensordict.html)

Building tensordicts from streams
![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys
![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule
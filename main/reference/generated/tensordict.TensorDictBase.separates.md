# TensorDictBase.separates

TensorDictBase.separates(**keys: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *default: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = NO_DEFAULT*, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *filter_empty: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self

Separates the specified keys from the tensordict in-place.

See also

This method is equivalent to calling [`split_keys()`](tensordict.TensorDictBase.split_keys.html#tensordict.TensorDictBase.split_keys) with
`inplace=True` on a single split.

See also

This method is equivalent to calling [`exclude()`](tensordict.TensorDictBase.exclude.html#tensordict.TensorDictBase.exclude) except that it
returns the other split of the data.

Parameters:

- **keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys to separate from the tensordict.
- **default** (*Any**,**optional*) - the value to be returned when a key is missing.
If not specified and `strict=True`, an exception is raised. Otherwise, the default of any missing key
will be `None` unless specified otherwise.
- **strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, an exception is raised when a key
is missing. Defaults to `True`.
- **filter_empty** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, empty tensordicts within `self` will be removed.
Defaults to `True`.

Returns:

the separated tensordict.

Return type:

T

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict(
... a=0,
... b=0,
... c=0,
... d=0,
... )
>>> td_a_c = td.separates("a", "c")
>>> print(td_a_c)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> print(td)
TensorDict(
 fields={
 b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
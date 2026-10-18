# TensorDictBase.separates

TensorDictBase.separates(**keys: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *default: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = NO_DEFAULT*, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *filter_empty: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self

Removes the entries of `keys` from the tensordict in place, and returns them in a new tensordict.

The values are not copied. The result keeps the nested tensordicts
that hold the removed entries, with the batch size of the tensordict.
This is [`split_keys()`](tensordict.TensorDictBase.split_keys.html#tensordict.TensorDictBase.split_keys) with `inplace=True` and one key set,
except that only the separated part is returned: the tensordict holds
the rest.

Parameters:

***keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys to remove.

Keyword Arguments:

- **default** (*Any**,**optional*) - if given, a missing key does not raise.
The value of `default` is not used: the result holds a
`None` entry under the missing key.
- **strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a missing key raises a
`KeyError`. If `False`, the result holds a `None` entry
under it. Defaults to `True`.
- **filter_empty** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the nested tensordicts
of the tensordict that the removal leaves empty are removed
too. Defaults to `True`.

Returns:

A new tensordict with the removed entries.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... a=torch.zeros(3),
... b=torch.zeros(3),
... nested=TensorDict(c=torch.zeros(3), batch_size=[3]),
... batch_size=[3],
... )
>>> out = td.separates("a", ("nested", "c"))
>>> sorted(out.keys(include_nested=True, leaves_only=True), key=str)
[('nested', 'c'), 'a']
>>> list(td.keys())
['b']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
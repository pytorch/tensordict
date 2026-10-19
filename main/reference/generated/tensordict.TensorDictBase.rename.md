# TensorDictBase.rename

TensorDictBase.rename(**names*, ***rename_map*)

Returns a shallow copy of the tensordict with new names for its batch dimensions.

Pass either the new names, one per batch dimension, or keyword
arguments that map old names to new ones. The leaves are not copied
(see [`copy()`](tensordict.TensorDictBase.copy.html#tensordict.TensorDictBase.copy)), and the names of the tensordict do not change.
The nested tensordicts of the copy get the new names for the batch
dimensions they share with it.

Parameters:

- ***names** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**None*) - the new names, one per batch dimension, or
a single `None` to clear every name. Exclusive with
`**rename_map`.
- ****rename_map** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**None*) - new names by old name, as in
`rename(old="new")`. Exclusive with `*names`.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if both or neither of `names` and `rename_map` are
 passed, or if a key of `rename_map` is not a name of the
 tensordict.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict(batch_size=[1, 2, 3, 4], names=list("abcd"))
>>> td.rename(c="g").names
['a', 'b', 'g', 'd']
>>> td.rename("w", "x", "y", "z").names
['w', 'x', 'y', 'z']
>>> td.rename(None).names
[None, None, None, None]
>>> td.names
['a', 'b', 'c', 'd']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
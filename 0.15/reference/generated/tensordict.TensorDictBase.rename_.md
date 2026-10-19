# TensorDictBase.rename

TensorDictBase.rename_(**names*, ***rename_map*)

Renames the batch dimensions of the tensordict in place and returns `self`.

Pass either the new names, one per batch dimension, or keyword
arguments that map old names to new ones. The nested tensordicts get
the new names for the batch dimensions they share with the tensordict.

Parameters:

- ***names** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**None*) - the new names, one per batch dimension, or
a single `None` to clear every name. Exclusive with
`**rename_map`.
- ****rename_map** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**None*) - new names by old name, as in
`rename_(old="new")`. Exclusive with `*names`.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if both `names` and `rename_map` are passed, or
 neither (except on a tensordict with no batch dimension), or if
 a key of `rename_map` is not a name of the tensordict.

See also

[`rename()`](tensordict.TensorDictBase.rename.html#tensordict.TensorDictBase.rename), which returns a renamed copy.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict(batch_size=[1, 2, 3, 4], names=list("abcd"))
>>> td.rename_(c="g") is td
True
>>> td.names
['a', 'b', 'g', 'd']
>>> td.rename_(None).names
[None, None, None, None]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
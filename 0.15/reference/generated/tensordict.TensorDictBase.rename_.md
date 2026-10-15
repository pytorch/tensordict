# TensorDictBase.rename

TensorDictBase.rename_(**names*, ***rename_map*)

Same as [`rename()`](tensordict.TensorDictBase.rename.html#tensordict.TensorDictBase.rename), but executes the renaming in-place.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict({}, batch_size=[1, 2, 3 ,4])
>>> td.names = list("abcd")
>>> td_renamed = td.rename_(c="g")
>>> assert td_renamed is td
>>> assert td.names == list("abgd")
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
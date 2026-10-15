# TensorDictBase.rename

TensorDictBase.rename(**names*, ***rename_map*)

Returns a clone of the tensordict with dimensions renamed.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict({}, batch_size=[1, 2, 3 ,4])
>>> td.names = list("abcd")
>>> td_rename = td.rename(c="g")
>>> assert td_rename.names == list("abgd")
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
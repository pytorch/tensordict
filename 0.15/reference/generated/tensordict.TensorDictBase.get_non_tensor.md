# TensorDictBase.get_non_tensor

TensorDictBase.get_non_tensor(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *default=NO_DEFAULT*)

Returns the content of the non-tensor entry at `key`.

A [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) returns its data, and a stack of
non-tensor data returns a list. An entry that is not non-tensor data
is returned as it is. See [Non-tensor data and unbatched entries](../../concepts/non_tensor.html#non-tensor-data).

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the entry.
- **default** (*Any**,**optional*) - the value returned if the key is missing.
If not passed, a missing key raises `KeyError`.

Raises:

[**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if the key is missing and no default is passed.

See also

[`set_non_tensor()`](tensordict.TensorDictBase.set_non_tensor.html#tensordict.TensorDictBase.set_non_tensor).

Examples

```
>>> from tensordict import TensorDict
>>> data = TensorDict({}, batch_size=[])
>>> data = data.set_non_tensor(("nested", "the string"), "a string!")
>>> data.get_non_tensor(("nested", "the string"))
'a string!'
>>> data.get_non_tensor("missing", "default")
'default'
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
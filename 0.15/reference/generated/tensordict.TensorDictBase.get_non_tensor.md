# TensorDictBase.get_non_tensor

TensorDictBase.get_non_tensor(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *default=NO_DEFAULT*)

Gets a non-tensor value, if it exists, or default if the non-tensor value is not found.

This method is robust to tensor/TensorDict values, meaning that if the
value gathered is a regular tensor it will be returned too (although
this method comes with some overhead and should not be used out of its
natural scope).

See [`set_non_tensor()`](tensordict.TensorDictBase.set_non_tensor.html#tensordict.TensorDictBase.set_non_tensor) for more information
on how to set non-tensor values in a tensordict.

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the location of the NonTensorData object.
- **default** (*Any**,**optional*) - the value to be returned if the key cannot
be found.

Returns: the content of the [`tensordict.tensorclass.NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData),

or the entry corresponding to the `key` if it isn't a
[`tensordict.tensorclass.NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) (or `default` if the
entry cannot be found).

Examples

```
>>> from tensordict import TensorDict
>>> data = TensorDict({}, batch_size=[])
>>> data = data.set_non_tensor(("nested", "the string"), "a string!")
>>> assert data.get_non_tensor(("nested", "the string")) == "a string!"
>>> # regular `get` works but returns a NonTensorData object
>>> data.get(("nested", "the string"))
NonTensorData(data=a string!, batch_size=torch.Size([]), device=None)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
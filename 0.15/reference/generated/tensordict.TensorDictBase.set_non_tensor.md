# TensorDictBase.set_non_tensor

TensorDictBase.set_non_tensor(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *value: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*)

Stores `value` as non-tensor data at `key`, and returns `self`.

The value is wrapped in a [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) with the
batch size, device and names of the tensordict that holds it. A tuple
key creates the missing nested tensordicts. Unlike [`set()`](tensordict.TensorDictBase.set.html#tensordict.TensorDictBase.set), a
number is stored as non-tensor data too. `td[key]` and
[`get_non_tensor()`](tensordict.TensorDictBase.get_non_tensor.html#tensordict.TensorDictBase.get_non_tensor) return `value`, and [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) returns
the `NonTensorData`. See [Non-tensor data and unbatched entries](../../concepts/non_tensor.html#non-tensor-data).

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the entry.
- **value** (*Any*) - the value to store.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the tensordict is locked.

Examples

```
>>> from tensordict import TensorDict
>>> data = TensorDict({}, batch_size=[])
>>> data = data.set_non_tensor(("nested", "the string"), "a string!")
>>> data["nested", "the string"]
'a string!'
>>> data.get(("nested", "the string"))
NonTensorData(data=a string!, batch_size=torch.Size([]), device=None)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
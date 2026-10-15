# TensorDictBase.set_non_tensor

TensorDictBase.set_non_tensor(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *value: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*)

Registers a non-tensor value in the tensordict using [`tensordict.tensorclass.NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData).

The value can be retrieved using [`TensorDictBase.get_non_tensor()`](tensordict.TensorDictBase.get_non_tensor.html#tensordict.TensorDictBase.get_non_tensor)
or directly using get, which will return the [`tensordict.tensorclass.NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData)
object.

return: self

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
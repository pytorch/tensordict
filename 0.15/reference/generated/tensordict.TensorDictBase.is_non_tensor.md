# TensorDictBase.is_non_tensor

TensorDictBase.is_non_tensor(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Checks if the value associated with a key is a non-tensor ([`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData)).

Unlike the standalone [`is_non_tensor()`](tensordict.is_non_tensor.html#tensordict.is_non_tensor) function which checks
a value directly, this method checks the **internal representation** of the
stored value, making it reliable for both [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) and
[`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass) instances.

This is useful because [`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass) `get()` unwraps
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) to plain values, so calling
`is_non_tensor(tc.get(key))` on the result would return `False`.
This method always checks the wrapped storage.

Parameters:

**key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key to check.

Returns:

`True` if the value stored at `key` is a
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) or [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack),
`False` otherwise.

Examples

```
>>> from tensordict import TensorDict
>>> import torch
>>> td = TensorDict({"obs": torch.randn(4)}, [])
>>> td = td.set_non_tensor("label", "cat")
>>> td.is_non_tensor("label")
True
>>> td.is_non_tensor("obs")
False
```

See also

[`set_non_tensor()`](tensordict.TensorDictBase.set_non_tensor.html#tensordict.TensorDictBase.set_non_tensor),
[`get_non_tensor()`](tensordict.TensorDictBase.get_non_tensor.html#tensordict.TensorDictBase.get_non_tensor).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
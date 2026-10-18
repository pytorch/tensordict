# TensorDictBase.is_non_tensor

TensorDictBase.is_non_tensor(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Returns `True` if the entry at `key` is non-tensor data.

Non-tensor data is a [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) or a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack). Unlike the function
[`is_non_tensor()`](tensordict.is_non_tensor.html#tensordict.is_non_tensor), which checks a value, this method
checks the stored entry. This matters for a tensorclass, whose
`get` returns the content of a non-tensor entry, not the
`NonTensorData`.

Parameters:

**key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key to check.

Raises:

[**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if the key is missing.

See also

[`set_non_tensor()`](tensordict.TensorDictBase.set_non_tensor.html#tensordict.TensorDictBase.set_non_tensor) and [`get_non_tensor()`](tensordict.TensorDictBase.get_non_tensor.html#tensordict.TensorDictBase.get_non_tensor).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"obs": torch.zeros(4)}, [])
>>> td = td.set_non_tensor("label", "cat")
>>> td.is_non_tensor("label")
True
>>> td.is_non_tensor("obs")
False
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
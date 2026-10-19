# TensorDictBase.non_tensor_items

TensorDictBase.non_tensor_items(*include_nested: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns a tuple of the `(key, value)` pairs of the non-tensor data of the tensordict.

The values are [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) (or
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack)) instances. See
[Non-tensor data and unbatched entries](../../concepts/non_tensor.html#non-tensor-data).

Parameters:

**include_nested** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the non-tensor data of
the nested tensordicts is also returned, under nested keys.
Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(2), s="text", nested=TensorDict(t="more"))
>>> [key for key, _ in td.non_tensor_items(include_nested=True)]
['s', ('nested', 't')]
>>> td.non_tensor_items()[0][1].data
'text'
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
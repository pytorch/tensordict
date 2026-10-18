# TensorDictBase.to_padded_tensor

TensorDictBase.to_padded_tensor(*padding=0.0*, *mask_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Returns a tensordict whose nested-tensor leaves are converted to padded tensors.

Calls [`torch.nested.to_padded_tensor()`](https://docs.pytorch.org/docs/stable/nested.html#torch.nested.to_padded_tensor) on every leaf that is a
nested tensor, and keeps the other leaves. If the batch size has a
`-1` (a ragged dimension, as in a dense version of a lazy stack
of tensordicts with different batch sizes), the batch size of the
result is computed from the padded leaves.

Parameters:

- **padding** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,**optional*) - the value of the padding. Defaults to
`0.0`.
- **mask_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - a key where a boolean mask of the
valid (not padded) elements is written. It needs a ragged batch
dimension, or a `RuntimeError` is raised. Defaults to
`None`, which writes no mask.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> stack = TensorDict.lazy_stack([
... TensorDict(a=torch.zeros(3), batch_size=[3]),
... TensorDict(a=torch.ones(4), batch_size=[4]),
... ])
>>> td = stack.densify(layout=torch.jagged)
>>> td.batch_size
torch.Size([2, -1])
>>> padded = td.to_padded_tensor(padding=-1.0, mask_key="mask")
>>> padded.batch_size
torch.Size([2, 4])
>>> padded["a"]
tensor([[ 0., 0., 0., -1.],
 [ 1., 1., 1., 1.]])
>>> padded["mask"]
tensor([[ True, True, True, False],
 [ True, True, True, True]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
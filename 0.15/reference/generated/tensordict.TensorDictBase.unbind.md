# TensorDictBase.unbind

TensorDictBase.unbind(*dim: [int](https://docs.python.org/3/builtins/functions.html#int)*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase), ...]

Returns a tuple of the tensordicts along a batch dimension, as [`torch.Tensor.unbind()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unbind.html#torch.Tensor.unbind) does.

The tensordicts of the result hold views of the original tensors. They
are locked if the tensordict is shared or memory-mapped.

Parameters:

**dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the batch dimension to remove. The result has
`batch_size[dim]` tensordicts.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... 'x': torch.arange(12).reshape(3, 4),
... }, batch_size=[3, 4])
>>> td0, td1, td2 = td.unbind(0)
>>> td0['x']
tensor([0, 1, 2, 3])
>>> td1['x']
tensor([4, 5, 6, 7])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `unbind`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
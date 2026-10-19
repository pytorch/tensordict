# TensorDictBase.permute

TensorDictBase.permute(**args*, ***kwargs*)

Returns a tensordict with its batch dimensions permuted, as [`torch.Tensor.permute()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.permute.html#torch.Tensor.permute) does.

The leaves keep their feature dimensions after the batch dimensions,
and are views of the original leaves. The names are permuted with the
dimensions.

It can be used as a context manager, as [`unsqueeze()`](tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze) can: when
the `with` block exits, the entries of the block are written back to
the tensordict with the inverse operation.

Parameters:

***dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the new order of the batch dimensions, as integers,
as one list or tuple, or as the keyword argument `dims`.

Returns:

A tensordict with the batch dimensions in the new order.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.zeros(3, 4, 5)}, [3, 4])
>>> td.permute(1, 0)["a"].shape
torch.Size([4, 3, 5])
>>> td.permute([1, 0]).batch_size
torch.Size([4, 3])
>>> td.permute(dims=[1, 0]).batch_size
torch.Size([4, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `permute`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
# TensorDictBase.squeeze

TensorDictBase.squeeze(**args*, ***kwargs*)

Returns a tensordict without the batch dimensions of size 1.

The tensors of the result are views of the original tensors. Feature
dimensions of size 1 are kept.

Parameters:

**dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch dimension to remove, in
`[-batch_dims, batch_dims - 1]`. If its size is not 1, the
tensordict is returned as it is. Defaults to `None`, which
removes every batch dimension of size 1.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... 'x': torch.arange(24).reshape(3, 1, 4, 2),
... }, batch_size=[3, 1, 4])
>>> td = td.squeeze()
>>> td.shape
torch.Size([3, 4])
>>> td.get("x").shape
torch.Size([3, 4, 2])
```

This operation can be used as a context manager too. Inside the
`with` block, you work on the squeezed tensordict. When the block
exits, the inverse operation ([`unsqueeze()`](tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze)) runs, and the
entries of the block are written back to the original tensordict. They
replace the original tensors instead of being copied into them. The
original tensordict must not be locked. If the block raises an
exception, nothing is written back. Pass `dim`: `with td.squeeze():`
raises `RuntimeError`, because the inverse operation needs the
dimension.

```
>>> td = TensorDict({
... 'x': torch.arange(24).reshape(3, 1, 4, 2),
... }, batch_size=[3, 1, 4])
>>> with td.squeeze(1) as tds:
... _ = tds.set("y", torch.zeros(3, 4))
>>> assert td.get("y").shape == (3, 1, 4)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `squeeze`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
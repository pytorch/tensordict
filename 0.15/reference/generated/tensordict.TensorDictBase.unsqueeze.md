# TensorDictBase.unsqueeze

TensorDictBase.unsqueeze(**args*, ***kwargs*)

Returns a tensordict with a new batch dimension of size 1 at position `dim`.

The tensors of the result are views of the original tensors. The new
dimension has no name.

Parameters:

**dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the position of the new dimension, in
`[-batch_dims - 1, batch_dims]` as for [`torch.unsqueeze()`](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html#torch.unsqueeze).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... 'x': torch.arange(24).reshape(3, 4, 2),
... }, batch_size=[3, 4])
>>> td = td.unsqueeze(-2)
>>> td.shape
torch.Size([3, 1, 4])
>>> td.get("x").shape
torch.Size([3, 1, 4, 2])
```

This operation can be used as a context manager too. Inside the
`with` block, you work on the unsqueezed tensordict. When the block
exits, the inverse operation ([`squeeze()`](tensordict.TensorDictBase.squeeze.html#tensordict.TensorDictBase.squeeze)) runs, and the entries
of the block are written back to the original tensordict. They replace
the original tensors instead of being copied into them. The original
tensordict must not be locked. If the block raises an exception,
nothing is written back.

```
>>> td = TensorDict({
... 'x': torch.arange(24).reshape(3, 4, 2),
... }, batch_size=[3, 4])
>>> with td.unsqueeze(-2) as tds:
... _ = tds.set("y", torch.zeros(3, 1, 4))
>>> assert td.get("y").shape == (3, 4)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `unsqueeze`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
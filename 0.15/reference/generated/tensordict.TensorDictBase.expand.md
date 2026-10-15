# TensorDictBase.expand

TensorDictBase.expand(**args*, ***kwargs*) → Self

Expands each tensor of the tensordict according to [`torch.Tensor.expand()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand), ignoring the feature dimensions.

The new shape is the new batch size. As for [`torch.Tensor.expand()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand),
it can add leading dimensions and expand the dimensions of size 1. The
other dimensions must keep their size. The result holds views of the
original tensors: no memory is allocated, and the expanded leaves are
not contiguous. Call [`clone()`](tensordict.TensorDictBase.clone.html#tensordict.TensorDictBase.clone) or [`contiguous()`](tensordict.TensorDictBase.contiguous.html#tensordict.TensorDictBase.contiguous) before
you write into them.

Supports iterables to specify the shape.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... 'a': torch.zeros(3, 4, 5),
... 'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
>>> td_expand = td.expand(10, 3, 4)
>>> assert td_expand.shape == torch.Size([10, 3, 4])
>>> assert td_expand.get("a").shape == torch.Size([10, 3, 4, 5])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `expand`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
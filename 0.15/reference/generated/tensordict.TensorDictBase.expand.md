# TensorDictBase.expand

TensorDictBase.expand(**args*, ***kwargs*) → Self

Returns a tensordict whose batch dimensions are expanded to `shape`.

The new shape is the new batch size, and the leaves keep their feature
dimensions. As for [`torch.Tensor.expand()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand), it can add leading
dimensions and expand the dimensions of size 1, and `-1` keeps the
size of a dimension. The other dimensions must keep their size. The
new leading dimensions have no name.

The result holds views of the original tensors: no memory is
allocated, and the expanded leaves are not contiguous. Call
[`clone()`](tensordict.TensorDictBase.clone.html#tensordict.TensorDictBase.clone) or [`contiguous()`](tensordict.TensorDictBase.contiguous.html#tensordict.TensorDictBase.contiguous) before you write into them.

Parameters:

***shape** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the new batch size, as integers or as one
`torch.Size`, list or tuple.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if `shape` has fewer dimensions than the batch
 size, or changes the size of a dimension that is not 1.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... 'a': torch.zeros(3, 4, 5),
... 'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
>>> td_expand = td.expand(10, 3, 4)
>>> td_expand.batch_size, td_expand["a"].shape
(torch.Size([10, 3, 4]), torch.Size([10, 3, 4, 5]))
>>> td.expand(2, -1, -1).batch_size
torch.Size([2, 3, 4])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `expand`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
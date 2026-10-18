# TensorDictBase.gather

TensorDictBase.gather(*dim: [int](https://docs.python.org/3/builtins/functions.html#int)*, *index: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *out: T | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Gathers values along a batch dimension, as [`torch.gather()`](https://docs.pytorch.org/docs/stable/generated/torch.gather.html#torch.gather) does.

`index` has as many dimensions as the batch, and gives the batch size
of the result; each leaf keeps its feature dimensions. As for
[`torch.gather()`](https://docs.pytorch.org/docs/stable/generated/torch.gather.html#torch.gather), `index` can be smaller than the batch along
the other dimensions. The data is copied, and the names are kept.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the batch dimension to gather along.
- **index** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a long tensor of indices along `dim`, with
as many dimensions as the batch.
- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict to write the result
into, with the batch size `index.shape`. Exclusive with
`inplace`. Defaults to `None`.

Keyword Arguments:

**inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict keeps its
identity and keys, its leaves are replaced one by one, and the
call returns `self`. It needs `index.ndim == batch_dims`.
Exclusive with `out`. A lazy stack, or a tensordict that
holds one, raises a `NotImplementedError`: call
[`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) first. See [In-place operations](../../concepts/operations.html#inplace-conventions).
Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... a=torch.arange(12).view(3, 4),
... b=TensorDict(c=torch.arange(12).view(3, 4, 1), batch_size=[3, 4, 1]),
... batch_size=[3, 4],
... )
>>> index = torch.tensor([[0, 3], [1, 2], [3, 3]])
>>> out = td.gather(dim=1, index=index)
>>> out["a"]
tensor([[ 0, 3],
 [ 5, 6],
 [11, 11]])
>>> out.batch_size, out["b", "c"].shape
(torch.Size([3, 2]), torch.Size([3, 2, 1]))
```

The names of the batch dimensions are kept:

```
>>> td.names = ["x", "y"]
>>> td.gather(dim=1, index=index).names
['x', 'y']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `gather`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
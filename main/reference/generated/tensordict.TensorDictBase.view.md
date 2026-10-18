# TensorDictBase.view

TensorDictBase.view(**shape: [int](https://docs.python.org/3/builtins/functions.html#int)*, *size: [list](https://docs.python.org/3/builtins/stdtypes.html#list) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple) | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Returns a tensordict with a new batch size, whose leaves are views of the original leaves.

Each leaf keeps its feature dimensions and shares storage with the
original leaf. As for [`torch.Tensor.view()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view), the leaves must have a
compatible memory layout: otherwise, use [`reshape()`](tensordict.TensorDictBase.reshape.html#tensordict.TensorDictBase.reshape). The result
has no names, and the tensordict itself is returned when the batch
size does not change.

`view(dtype)` views every leaf with another dtype instead, as
[`torch.Tensor.view()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view) does. The last dimension of a leaf changes
size when the dtypes have different sizes, so the leaves then need a
feature dimension.

It can be used as a context manager, as [`unsqueeze()`](tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze) can: when
the `with` block exits, the entries of the block are written back to
the tensordict with the inverse operation.

Parameters:

***shape** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the new batch size, as integers or as one
`torch.Size`, list or tuple. One dimension can be `-1`.
Alternatively, a `torch.dtype`.

Keyword Arguments:

- **size** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*,*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*or*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the new batch size,
as a keyword argument instead of `*shape`. Defaults to
`None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - with a dtype, the batch size of
the result. It is not used with a shape. Defaults to `None`,
which keeps the batch size.

Returns:

A tensordict with the new batch size.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(source={'a': torch.zeros(3,4,5),
... 'b': torch.zeros(3,4,10,1)}, batch_size=torch.Size([3, 4]))
>>> td_view = td.view(12)
>>> print(td_view.get("a").shape)
torch.Size([12, 5])
>>> print(td_view.get("b").shape)
torch.Size([12, 10, 1])
>>> td_view = td.view(-1, 4, 3)
>>> print(td_view.get("a").shape)
torch.Size([1, 4, 3, 5])
>>> print(td_view.get("b").shape)
torch.Size([1, 4, 3, 10, 1])
>>> td_int = td.view(torch.int32)
>>> td_int["a"].dtype, td_int.batch_size
(torch.int32, torch.Size([3, 4]))
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `view`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
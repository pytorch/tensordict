# TensorDictBase.view

TensorDictBase.view(**shape: [int](https://docs.python.org/3/builtins/functions.html#int)*, *size: [list](https://docs.python.org/3/builtins/stdtypes.html#list) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple) | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Returns a tensordict with views of the tensors according to a new shape, compatible with the tensordict batch_size.

Each leaf keeps its feature dimensions, and is a view of the original
leaf: they share storage. As for [`torch.Tensor.view()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view), the
leaves must have a compatible memory layout. Otherwise, use
[`reshape()`](tensordict.TensorDictBase.reshape.html#tensordict.TensorDictBase.reshape).

Alternatively, a dtype can be provided as a first unnamed argument. In that case, all tensors will be viewed
with the according dtype. Note that this assume that the new shapes will be compatible with the provided dtype.
See [`torch.Tensor.view()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view) for more information on dtype views.

Parameters:

- ***shape** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - new batch size of the resulting tensordict, as
integers or as one `torch.Size`.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - alternatively, a dtype to use to represent the tensor content.

Keyword Arguments:

- **size** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*,*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*or*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the new batch size,
as a keyword argument instead of `*shape`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - if a dtype is provided, the batch-size can be reset using this
keyword argument. If the `view` is called with a shape, this is without effect.

Returns:

a new tensordict with the desired batch_size.

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
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `view`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
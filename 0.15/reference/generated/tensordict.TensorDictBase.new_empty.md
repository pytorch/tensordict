# TensorDictBase.new_empty

TensorDictBase.new_empty(**size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) = NO_DEFAULT*, *requires_grad: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *layout: [layout](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.layout) = torch.strided*, *pin_memory: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *empty_lazy: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns a tensordict of batch size `size` with uninitialized leaves.

The values of the leaves are not set, as with [`torch.empty()`](https://docs.pytorch.org/docs/stable/generated/torch.empty.html#torch.empty).

Each leaf keeps its feature dimensions: a leaf of shape
`(*self.batch_size, *f)` becomes `(*size, *f)`. By default, each
leaf keeps its dtype and device.

Parameters:

***size** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or**sequence**of*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - the batch size of the result, as
integers or as one list, tuple or `torch.Size`.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype of every leaf of the
result. Defaults to `None`: each leaf keeps its dtype.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result and of its leaves. If not passed, each leaf keeps its
device, and the result keeps the device of the tensordict.
- **requires_grad** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, autograd records the
operations on the new leaves. Defaults to `False`.
- **layout** ([*layout*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.layout)*,**optional*) - the layout of the new leaves.
Defaults to `torch.strided`.
- **pin_memory** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the new leaves are
allocated in pinned memory. Works only for CPU tensors.
Defaults to `False`.
- **empty_lazy** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the lazy stacks of the
result are emptied of their content. Use it when the content
of a lazy stack is likely to change while you fill the new
tensordict. It applies to the sub-tensordicts too. Defaults to
`False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3, 4), b=torch.zeros(3, dtype=torch.int64), batch_size=[3])
>>> out = td.new_empty(5)
>>> out.batch_size, out["a"].shape
(torch.Size([5]), torch.Size([5, 4]))
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
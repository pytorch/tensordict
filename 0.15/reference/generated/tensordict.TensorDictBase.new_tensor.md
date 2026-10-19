# TensorDictBase.new_tensor

TensorDictBase.new_tensor(*data: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence) | [float](https://docs.python.org/3/builtins/functions.html#float)*, ***, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) = NO_DEFAULT*, *requires_grad: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *pin_memory: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Returns a tensordict whose leaves are copies of `data`, each with the dtype of its leaf.

Each leaf of the tensordict gives a leaf of the result, built as
`leaf.new_tensor(data)`: the batch size of the result and the shape
of every leaf are the shape of `data`. If `data` is a tensordict or
a tensorclass, each leaf is built from the matching leaf of `data`.

Parameters:

**data** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**array_like*) - the data to
copy. As for [`torch.Tensor.new_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.new_tensor.html#torch.Tensor.new_tensor), it can also be a
list, a tuple, a NumPy array or a scalar.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype of every leaf of the
result. Defaults to `None`: each leaf keeps the dtype of the
matching leaf of the tensordict.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result and of its leaves. If not passed, the device of
`data`: the default device for a list, a tuple, a NumPy array
or a scalar.
- **requires_grad** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, autograd records the
operations on the new leaves. Defaults to `False`.
- **pin_memory** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the new leaves are
allocated in pinned memory. Works only for CPU tensors.
Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... a=torch.zeros(3), b=torch.zeros(3, dtype=torch.int64), batch_size=[3]
... )
>>> out = td.new_tensor([1, 2])
>>> out.batch_size
torch.Size([2])
>>> out["a"], out["b"]
(tensor([1., 2.]), tensor([1, 2]))
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
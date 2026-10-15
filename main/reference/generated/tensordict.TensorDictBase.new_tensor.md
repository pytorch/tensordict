# TensorDictBase.new_tensor

TensorDictBase.new_tensor(*data: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence) | [float](https://docs.python.org/3/builtins/functions.html#float)*, ***, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) = NO_DEFAULT*, *requires_grad: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *pin_memory: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Returns a new TensorDict with data as the tensor `data`.

By default, each entry of the returned TensorDict has the `torch.dtype` of the
matching entry of this tensordict, and the returned TensorDict is on the device
of `data`.

The `data` can also be a tensor collection (`TensorDict` or `tensorclass`), in which case
the `new_tensor` method iterates over the tensor pairs of `self` and `data`.

Parameters:

**data** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**array_like*) - the data to be copied. As
for [`torch.Tensor.new_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.new_tensor.html#torch.Tensor.new_tensor), it can be a list, a tuple, a NumPy
array or a scalar; such data has no device, so the result is on the
device that [`torch.as_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.as_tensor.html#torch.as_tensor) gives it, the default device.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the desired type of returned tensordict.
Default: if `None`, each entry keeps the `torch.dtype` of the
matching entry of this tensordict.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the desired device of returned tensordict.
Default: the device of `data`.
- **requires_grad** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If autograd should record operations on the
returned tensors. Default: `False`.
- **pin_memory** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If set, returned tensor would be allocated in the
pinned memory. Works only for CPU tensors. Default: `False`.

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
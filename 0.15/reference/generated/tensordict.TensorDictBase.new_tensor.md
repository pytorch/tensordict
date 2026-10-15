# TensorDictBase.new_tensor

TensorDictBase.new_tensor(*data: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection*, ***, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) = NO_DEFAULT*, *requires_grad: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *pin_memory: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Returns a new TensorDict with data as the tensor `data`.

By default, the returned TensorDict values have the same `torch.dtype` and `torch.device` as this tensor.

The `data` can also be a tensor collection (`TensorDict` or `tensorclass`), in which case
the `new_tensor` method iterates over the tensor pairs of `self` and `data`.

Parameters:

**data** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the data to be copied.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the desired type of returned tensordict.
Default: if `None`, the torch.dtype will be unchanged.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the desired device of returned tensordict.
Default: if `None`, the `torch.device` will be unchanged.
- **requires_grad** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If autograd should record operations on the
returned tensors. Default: `False`.
- **pin_memory** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If set, returned tensor would be allocated in the
pinned memory. Works only for CPU tensors. Default: `False`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
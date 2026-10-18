# TensorDictBase.from_tuple

*classmethod*TensorDictBase.from_tuple(*obj*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Converts a tuple to a tensordict with the keys `"0"`, `"1"`, ...

Each element is converted with [`from_any()`](tensordict.TensorDictBase.from_any.html#tensordict.TensorDictBase.from_any), so a nested tuple
or dict becomes a nested tensordict and a number becomes a tensor.

Parameters:

**obj** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) - the tuple to convert.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`. Defaults to `None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
It applies to the nested tensordicts too. Exclusive with
`auto_batch_size`. Defaults to `None`, an empty batch size.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if both `auto_batch_size=True` and `batch_size`
 are given.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict.from_tuple((torch.zeros(3, 4), 1), auto_batch_size=True)
>>> td["0"].shape, td["1"], td.batch_size
(torch.Size([3, 4]), tensor(1), torch.Size([]))
>>> td = TensorDict.from_tuple((torch.zeros(3, 4), torch.ones(3)), auto_batch_size=True)
>>> td.batch_size
torch.Size([3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
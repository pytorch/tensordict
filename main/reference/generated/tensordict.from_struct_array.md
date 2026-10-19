# from_struct_array

tensordict.from_struct_array(*struct_array*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Converts a NumPy structured array to a tensordict, with one entry per field.

The tensors share memory with the array. See
[`TensorDictBase.from_struct_array`](tensordict.TensorDictBase.from_struct_array.html#tensordict.TensorDictBase.from_struct_array).

Parameters:

**struct_array** (*np.ndarray*) - the structured array to convert.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size to
the largest leading shape that all the entries share. Exclusive
with `batch_size`. Defaults to `False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. `None` leaves
each leaf where it is, and the result has no device. Defaults to
`None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`, which
uses the shape of the array.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).

Examples

```
>>> import numpy as np
>>> from tensordict import from_struct_array
>>> x = np.array(
... [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
... dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
... )
>>> td = from_struct_array(x)
>>> td.batch_size, td["age"]
(torch.Size([2]), tensor([9, 3], dtype=torch.int32))
>>> x["age"] += 1 # the tensordict shares memory with the array
>>> td["age"]
tensor([10, 4], dtype=torch.int32)
>>> bool((td.to_struct_array() == x).all())
True
```
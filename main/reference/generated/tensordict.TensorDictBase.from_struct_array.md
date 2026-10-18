# TensorDictBase.from_struct_array

*classmethod*TensorDictBase.from_struct_array(*struct_array: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Converts a NumPy structured array to a tensordict, with one entry per field.

The batch size is the shape of the array, and each entry holds a
field. The conversion is zero-copy: the tensors share memory with the
array, so an in-place change to the array changes the tensordict, and
the other way around. Two cases copy the data: a `device` other
than the CPU, and a field whose item size does not divide the size of
a record, such as the `"i8"` field of a packed array of `"i8"` and
`"f4"` fields (12 bytes per record), which torch cannot view. String
fields become non-tensor data. [`to_struct_array()`](tensordict.TensorDictBase.to_struct_array.html#tensordict.TensorDictBase.to_struct_array) converts
back.

Parameters:

**struct_array** (*np.ndarray*) - the structured array to convert.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share,
which can add the dimensions of array fields. Exclusive with
`batch_size`. Defaults to `False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`. Defaults to `None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`,
which uses the shape of the array.

Returns:

A tensordict of the class this method is called on
([`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) for `TensorDictBase`).

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if both `auto_batch_size=True` and `batch_size`
 are given.

Examples

```
>>> import numpy as np
>>> from tensordict import TensorDict
>>> x = np.array(
... [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
... dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
... )
>>> td = TensorDict.from_struct_array(x)
>>> td.batch_size, td["age"]
(torch.Size([2]), tensor([9, 3], dtype=torch.int32))
>>> x["age"] += 1 # the tensordict shares memory with the array
>>> td["age"]
tensor([10, 4], dtype=torch.int32)
>>> bool((td.to_struct_array() == x).all())
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).
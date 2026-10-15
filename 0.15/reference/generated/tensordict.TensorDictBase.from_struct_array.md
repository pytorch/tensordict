# TensorDictBase.from_struct_array

*classmethod*TensorDictBase.from_struct_array(*struct_array: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Converts a structured numpy array to a TensorDict.

The resulting TensorDict will share the same memory content as the numpy array (it is a zero-copy operation).
Changing values of the structured numpy array in-place will affect the content of the TensorDict.

Note

This method performs a zero-copy operation, meaning that the resulting TensorDict will share the same memory
content as the input numpy array. Therefore, changing values of the numpy array in-place will affect the content
of the TensorDict.

Parameters:

**struct_array** (*np.ndarray*) - The structured numpy array to be converted.

Keyword Arguments:

- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, the batch size will be computed automatically. Defaults to `False`.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - If `auto_batch_size` is `True`, defines how many dimensions the output
tensordict should have. Defaults to `None` (full batch-size at each level).
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - 

The device on which the TensorDict will be created.
Defaults to `None`.

Note

Changing the device (i.e., specifying any device other than `None` or `"cpu"`) will transfer the data,
resulting in a change to the memory location of the returned data.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict. Defaults to None.

Returns:

A TensorDict representation of the input structured numpy array.

Examples

```
>>> import numpy as np
>>> from tensordict import TensorDict
>>> x = np.array(
... [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
... dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
... )
>>> td = TensorDict.from_struct_array(x)
>>> x_recon = td.to_struct_array()
>>> assert (x_recon == x).all()
>>> assert x_recon.shape == x.shape
>>> # Try modifying x age field and check effect on td
>>> x["age"] += 1
>>> assert (td["age"] == np.array([10, 4])).all()
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).